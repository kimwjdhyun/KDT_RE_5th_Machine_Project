#include <Wire.h>
#include <DHT.h>
#include <SoftwareSerial.h>
#include <Adafruit_INA219.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DHT_PIN         5
#define DHT_TYPE        DHT11
#define LDR_PIN         A1
#define SOIL_PIN        A2

// UNO D7 <- ESP TX
// UNO D6 -> ESP RX (분압 권장)
#define ESP_RX_PIN      7
#define ESP_TX_PIN      6

#define WIFI_SSID       "JEONGHYUN 7867"
#define WIFI_PASS       "t=29790D"

#define SERVER_IP       "192.168.201.138"
#define SERVER_PORT     5000

#define SEND_INTERVAL   10000UL

#define BAT_MAX_V       16.8
#define BAT_MIN_V       12.0

DHT dht(DHT_PIN, DHT_TYPE);
SoftwareSerial espSerial(ESP_RX_PIN, ESP_TX_PIN);
Adafruit_INA219 ina219_solar(0x40);
Adafruit_INA219 ina219_battery(0x41);

unsigned long lastSendTime = 0;
unsigned long lastPrintTime = 0;
bool wifiOK = false;

// ----------------------------
// 유틸
// ----------------------------
int calcLight() {
  return map(analogRead(LDR_PIN), 0, 1023, 0, 100);
}

int calcSoil() {
  int raw = constrain(analogRead(SOIL_PIN), 300, 1023);
  return map(raw, 300, 1023, 100, 0);
}

int estimateSOC(float voltage) {
  float soc = (voltage - BAT_MIN_V) / (BAT_MAX_V - BAT_MIN_V) * 100.0;
  if (soc < 0) soc = 0;
  if (soc > 100) soc = 100;
  return (int)soc;
}

float sanitizeFloat(float value) {
  if (isnan(value) || isinf(value)) return 0.0;
  return value;
}

void clearESPBuffer() {
  while (espSerial.available()) {
    espSerial.read();
  }
}

bool waitForKeyword(const char* keyword, unsigned long timeoutMs) {
  const size_t BUF_SIZE = 120;
  char buf[BUF_SIZE];
  size_t idx = 0;
  unsigned long start = millis();

  memset(buf, 0, sizeof(buf));

  while (millis() - start < timeoutMs) {
    while (espSerial.available()) {
      char c = espSerial.read();

      if (idx < BUF_SIZE - 1) {
        buf[idx++] = c;
        buf[idx] = '\0';
      } else {
        memmove(buf, buf + 1, BUF_SIZE - 2);
        buf[BUF_SIZE - 2] = c;
        buf[BUF_SIZE - 1] = '\0';
      }

      if (strstr(buf, keyword) != NULL) {
        return true;
      }
    }
  }
  return false;
}

void readESPResponse(char* out, size_t outSize, unsigned long timeoutMs) {
  size_t idx = 0;
  unsigned long start = millis();

  if (outSize == 0) return;
  out[0] = '\0';

  while (millis() - start < timeoutMs) {
    while (espSerial.available()) {
      char c = espSerial.read();

      if (idx < outSize - 1) {
        out[idx++] = c;
        out[idx] = '\0';
      } else {
        memmove(out, out + 1, outSize - 2);
        out[outSize - 2] = c;
        out[outSize - 1] = '\0';
      }
    }
  }
}

bool sendATWait(const __FlashStringHelper* cmd, const char* keyword, unsigned long timeoutMs) {
  clearESPBuffer();
  espSerial.println(cmd);
  return waitForKeyword(keyword, timeoutMs);
}

bool sendATWaitRAM(const char* cmd, const char* keyword, unsigned long timeoutMs) {
  clearESPBuffer();
  espSerial.println(cmd);
  return waitForKeyword(keyword, timeoutMs);
}

// ----------------------------
// WiFi 연결
// ----------------------------
bool connectWifi() {
  Serial.println(F("[WiFi] 연결 시도"));

  if (!sendATWait(F("AT"), "OK", 2000)) {
    Serial.println(F("[WiFi] 실패: AT"));
    return false;
  }

  if (!sendATWait(F("AT+RST"), "ready", 5000)) {
    Serial.println(F("[WiFi] 실패: RST"));
    return false;
  }

  if (!sendATWait(F("AT+CWMODE=1"), "OK", 3000)) {
    Serial.println(F("[WiFi] 실패: CWMODE"));
    return false;
  }

  if (!sendATWait(F("AT+CIPMUX=0"), "OK", 3000)) {
    Serial.println(F("[WiFi] 실패: CIPMUX"));
    return false;
  }

  char joinCmd[96];
  snprintf(joinCmd, sizeof(joinCmd), "AT+CWJAP=\"%s\",\"%s\"", WIFI_SSID, WIFI_PASS);

  if (!sendATWaitRAM(joinCmd, "OK", 20000)) {
    Serial.println(F("[WiFi] 실패: CWJAP"));
    return false;
  }

  if (!sendATWait(F("AT+CIFSR"), "STAIP", 5000)) {
    Serial.println(F("[WiFi] 실패: CIFSR"));
    return false;
  }

  Serial.println(F("[WiFi] 연결 성공"));
  return true;
}

// ----------------------------
// 서버 전송
// ----------------------------
void sendToServer(
  float temperature,
  float humidity,
  int soil,
  int light,
  float solar_voltage_V,
  float solar_current_A,
  float solar_power_W,
  float battery_voltage_V,
  float battery_current_A,
  float battery_power_W,
  int soc
) {
  if (!wifiOK) {
    Serial.println(F("[WiFi] 재연결 시도"));
    wifiOK = connectWifi();
    if (!wifiOK) {
      Serial.println(F("[전송 스킵] WiFi 실패"));
      return;
    }
  }

  char json[256];
  int jsonLen = snprintf(
    json, sizeof(json),
    "{\"temperature\":%.1f,"
    "\"humidity\":%.1f,"
    "\"soil\":%d,"
    "\"light\":%d,"
    "\"solar_voltage\":%.2f,"
    "\"solar_current\":%.3f,"
    "\"solar_power\":%.3f,"
    "\"battery_voltage\":%.2f,"
    "\"battery_current\":%.3f,"
    "\"battery_power\":%.3f,"
    "\"pump\":0,"
    "\"led\":0,"
    "\"soc\":%d}",
    sanitizeFloat(temperature),
    sanitizeFloat(humidity),
    soil,
    light,
    sanitizeFloat(solar_voltage_V),
    sanitizeFloat(solar_current_A),
    sanitizeFloat(solar_power_W),
    sanitizeFloat(battery_voltage_V),
    sanitizeFloat(battery_current_A),
    sanitizeFloat(battery_power_W),
    soc
  );

  if (jsonLen <= 0 || jsonLen >= (int)sizeof(json)) {
    Serial.println(F("[오류] JSON 버퍼 부족"));
    return;
  }

  char req[420];
  int reqLen = snprintf(
    req, sizeof(req),
    "POST /sensor HTTP/1.1\r\n"
    "Host: %s:%d\r\n"
    "Content-Type: application/json\r\n"
    "Connection: close\r\n"
    "Content-Length: %d\r\n"
    "\r\n"
    "%s",
    SERVER_IP, SERVER_PORT, jsonLen, json
  );

  if (reqLen <= 0 || reqLen >= (int)sizeof(req)) {
    Serial.println(F("[오류] HTTP 버퍼 부족"));
    return;
  }

  Serial.println(F("[POST] 전송 시작"));

  sendATWait(F("AT+CIPCLOSE"), "OK", 500);

  char cipstartCmd[64];
  snprintf(cipstartCmd, sizeof(cipstartCmd),
           "AT+CIPSTART=\"TCP\",\"%s\",%d",
           SERVER_IP, SERVER_PORT);

  clearESPBuffer();
  espSerial.println(cipstartCmd);

  char cipRes[180];
  readESPResponse(cipRes, sizeof(cipRes), 5000);

  if (strstr(cipRes, "OK") == NULL &&
      strstr(cipRes, "CONNECT") == NULL &&
      strstr(cipRes, "ALREADY CONNECTED") == NULL) {
    Serial.println(F("[오류] TCP 연결 실패"));
    wifiOK = false;
    sendATWait(F("AT+CIPCLOSE"), "OK", 500);
    return;
  }

  char cipsendCmd[24];
  snprintf(cipsendCmd, sizeof(cipsendCmd), "AT+CIPSEND=%d", reqLen);

  clearESPBuffer();
  espSerial.println(cipsendCmd);

  if (!waitForKeyword(">", 3000)) {
    Serial.println(F("[오류] CIPSEND 실패"));
    wifiOK = false;
    sendATWait(F("AT+CIPCLOSE"), "OK", 500);
    return;
  }

  espSerial.print(req);

  char serverRes[320];
  readESPResponse(serverRes, sizeof(serverRes), 6000);

  if (strstr(serverRes, "SEND OK") != NULL || strstr(serverRes, "HTTP/1.1 200") != NULL) {
    Serial.println(F("[POST] 성공"));
  } else {
    Serial.println(F("[POST] 실패/불명확"));
    wifiOK = false;
  }

  sendATWait(F("AT+CIPCLOSE"), "OK", 500);
}

void setup() {
  Serial.begin(9600);
  espSerial.begin(9600);

  dht.begin();
  Wire.begin();

  Serial.println(F("[MASTER] 시작"));

  if (ina219_solar.begin()) {
    Serial.println(F("[INA219] solar OK"));
  } else {
    Serial.println(F("[INA219] solar FAIL"));
  }

  if (ina219_battery.begin()) {
    Serial.println(F("[INA219] battery OK"));
  } else {
    Serial.println(F("[INA219] battery FAIL"));
  }

  // 예전 흐름: setup에서 바로 WiFi 연결
  wifiOK = connectWifi();

  lastSendTime = millis() - SEND_INTERVAL;
  lastPrintTime = 0;
}

void loop() {
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  if (isnan(temperature)) temperature = 0.0;
  if (isnan(humidity)) humidity = 0.0;

  int light = calcLight();
  int soil = calcSoil();

  // 태양광 INA219
  float solar_busV = sanitizeFloat(ina219_solar.getBusVoltage_V());
  float solar_shuntV = sanitizeFloat(ina219_solar.getShuntVoltage_mV());
  float solar_current_mA = sanitizeFloat(ina219_solar.getCurrent_mA());
  float solar_power_mW = sanitizeFloat(ina219_solar.getPower_mW());

  float solar_voltage_V = sanitizeFloat(solar_busV + (solar_shuntV / 1000.0));
  float solar_current_A = sanitizeFloat(solar_current_mA / 1000.0);
  float solar_power_W = sanitizeFloat(solar_power_mW / 1000.0);

  // 배터리 INA219
  float battery_busV = sanitizeFloat(ina219_battery.getBusVoltage_V());
  float battery_shuntV = sanitizeFloat(ina219_battery.getShuntVoltage_mV());
  float battery_current_mA = sanitizeFloat(ina219_battery.getCurrent_mA());
  float battery_power_mW = sanitizeFloat(ina219_battery.getPower_mW());

  float battery_voltage_V = sanitizeFloat(battery_busV + (battery_shuntV / 1000.0));
  float battery_current_A = sanitizeFloat(battery_current_mA / 1000.0);
  float battery_power_W = sanitizeFloat(battery_power_mW / 1000.0);

  int soc = estimateSOC(battery_voltage_V);

  if (millis() - lastPrintTime >= 10000UL) {
    lastPrintTime = millis();

    Serial.print(F("T="));
    Serial.print(temperature, 1);
    Serial.print(F(" H="));
    Serial.print(humidity, 1);
    Serial.print(F(" L="));
    Serial.print(light);
    Serial.print(F(" S="));
    Serial.print(soil);

    Serial.print(F(" SV="));
    Serial.print(solar_voltage_V, 2);
    Serial.print(F(" SI="));
    Serial.print(solar_current_A, 3);
    Serial.print(F(" SP="));
    Serial.print(solar_power_W, 3);

    Serial.print(F(" BV="));
    Serial.print(battery_voltage_V, 2);
    Serial.print(F(" BI="));
    Serial.print(battery_current_A, 3);
    Serial.print(F(" BP="));
    Serial.print(battery_power_W, 3);
    Serial.print(F(" SOC="));
    Serial.println(soc);
  }

  if (millis() - lastSendTime >= SEND_INTERVAL) {
    lastSendTime = millis();

    sendToServer(
      temperature,
      humidity,
      soil,
      light,
      solar_voltage_V,
      solar_current_A,
      solar_power_W,
      battery_voltage_V,
      battery_current_A,
      battery_power_W,
      soc
    );
  }
}